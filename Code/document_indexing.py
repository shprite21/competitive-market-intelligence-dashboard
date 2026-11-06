# imports needed
import json
import os
import warnings
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from selection import validate_report_catalog


# code to load,chunk and embedd the document
class indexing_reports:
    def __init__(self, company_name, *, pinecone_api_key=None):
        self.company_name = company_name
        self.pinecone_api_key = pinecone_api_key
        self.code_dir = Path(__file__).resolve().parent
        self.project_dir = self.code_dir.parent
        self.report_data_path = self.code_dir / "report_data.json"
        self.reports_dir = self.project_dir / "Reports"

    @property
    def index_name(self):
        # Local nomic-embed-text vectors have 768 dimensions. Keep this separate
        # from the legacy OpenAI (1536 dimension) index so existing data is safe.
        return self.company_name.lower().replace(" ", "-") + "-local-v2"

    @property
    def embedding_model(self):
        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            # Release the embedding model before Qwen is asked to extract
            # results; both models may not fit in the available GPU memory.
            keep_alive=0,
        )

    def is_report_embedded(self, selected_reports: dict):
        # functioin to check if a report already embedded or not
        load_dotenv(Path(__file__).with_name(".env"), override=True)
        pc = Pinecone(
            api_key=self.pinecone_api_key
            if self.pinecone_api_key is not None
            else os.getenv("PINECONE_API_KEY")
        )
        index_name = self.index_name
        index_names = []
        for idx in pc.list_indexes():
            index_names.append(idx.name)
        if index_name not in index_names:
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            index = pc.Index(index_name)
            print("new index created")
            return selected_reports
        index = pc.Index(index_name)
        embedding_model = self.embedding_model
        relevant_reports = {}
        for name, url in selected_reports.items():
            response = index.query(
                vector=embedding_model.embed_query(self.company_name),
                filter={"report_name": {"$eq": name}},
                top_k=1,
            )
            matches = response.matches
            try:
                print(
                    f"Vector already embedded ->MATCH FOUND :{matches[0]['id']}->MOVING TO NEXT DOCUMENT "
                )
                continue
            except IndexError:
                relevant_reports[name] = url
        print(relevant_reports)

        return relevant_reports

    def load_reports(self, selected_reports: dict):
        # first filter out the reports that are already embedded
        relevant_reports = self.is_report_embedded(selected_reports)

        report_name_location = {}
        # opening json file to retrieve data
        with self.report_data_path.open("r", encoding="utf-8") as f:
            data = validate_report_catalog(json.load(f))

        for report in data:
            # checking if report is already downloaded on the system before
            if (
                report["company_name"] == self.company_name
                and report["report_name"] in relevant_reports
                and report.get("cached") is True
            ):
                if report["file_location"] is None:
                    warnings.warn(
                        f"{report['report_name']} file_location missing : unable to fetch report",
                        stacklevel=2,
                    )
                    report["cached"] = False
                else:
                    print(f"{report['report_name']} is already downloaded")
                    report_path = self._resolve_report_path(report["file_location"])
                    if not report_path.is_file():
                        warnings.warn(
                            f"Cached report is missing on disk: {report_path}",
                            UserWarning,
                            stacklevel=2,
                        )
                        report["cached"] = False
                    else:
                        report_name_location[report["report_name"]] = str(report_path)
            # if system is not downloaded before fetch it and then change its json attributes( cached=True, file_location : attach file location)
            if (
                report["company_name"] == self.company_name
                and report["report_name"] in relevant_reports
                and not report.get("cached", False)
            ):
                url = report["url"]
                if not url:
                    warnings.warn(
                        f"No download URL is configured for {report['report_name']}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                    response = requests.get(url, timeout=20, headers=headers)
                    response.raise_for_status()
                except requests.exceptions.RequestException as error:
                    print(f"Failed to retrieve {report['report_name']}: {error}")
                    warnings.warn(
                        "Failed to retrieve the report. The requested resource could not be accessed.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                if not response.content.startswith(b"%PDF"):
                    warnings.warn(
                        f"Downloaded content for {report['report_name']} is not a PDF",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                self.reports_dir.mkdir(parents=True, exist_ok=True)
                file_path = (
                    self.reports_dir / f"{report['company_name']}_{report['report_name']}.pdf"
                )
                file_path.write_bytes(response.content)

                report["cached"] = True
                report["file_location"] = str(Path("..") / "Reports" / file_path.name)
                report_name_location[report["report_name"]] = str(file_path)

                with self.report_data_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
        # loading for the document is completed . we now call the chunk documents function

        print(f"Documnet loading for company : {self.company_name} is completed")
        # before we call chunks document function we are going to check how many reports are already embedded in the vector store

        self.chunk_documents(report_name_location)

    def _resolve_report_path(self, file_location: str) -> Path:
        """Resolve catalog paths relative to this module, not the shell's cwd."""
        path = Path(file_location)
        if not path.is_absolute():
            path = self.code_dir / path
        return path.resolve()

    def chunk_documents(self, report_name_location: dict):
        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800, chunk_overlap=300, add_start_index=True
        )

        for name, loc in report_name_location.items():
            loader = PyPDFLoader(loc)
            document = loader.load()

            for doc in document:
                text = doc.page_content or ""

                # Default values
                section = "non-statement"

                # Check whether this page is one of the financial statements
                for statement_name in [
                    "Consolidated Statement of Profit and Loss",
                    "Consolidated Statement of Cash Flow",
                    "Consolidated Balance Sheet",
                ]:
                    if statement_name in text and "Notes to the " not in text:
                        section = statement_name
                        break

                # Update metadata once
                doc.metadata.update(
                    {
                        "company_name": self.company_name,
                        "report_name": name,
                        "text": doc.page_content,
                        "section": section,
                        "audited": name[0] == "A",
                    }
                )

            # convert to chunks

            chunks = text_splitter.split_documents(document)
            all_chunks.extend(chunks)

        print(f"All documents for the comapny :{self.company_name} are divided in chunks")
        print(len(all_chunks))
        self.embedd_and_store(all_chunks)

    def embedd_and_store(self, all_chunks: list):
        print("tt")

        embedding_model = self.embedding_model
        print("embedding model defined")
        # defining vector store

        pc = Pinecone(
            api_key=self.pinecone_api_key
            if self.pinecone_api_key is not None
            else os.getenv("PINECONE_API_KEY")
        )

        index = pc.Index(self.index_name)

        vector_store = PineconeVectorStore(embedding=embedding_model, index=index)

        print("Pinecone vector defined ")

        # Ollama runs embeddings locally. Sending every page chunk in a single
        # request can overwhelm its model runner, so upload predictable batches.
        batch_size = 32
        document_ids = []
        for start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[start : start + batch_size]
            document_ids.extend(vector_store.add_documents(batch))
            print(
                f"Embedded {min(start + batch_size, len(all_chunks))} of {len(all_chunks)} chunks"
            )

        print("all chunks documented")

        return document_ids
