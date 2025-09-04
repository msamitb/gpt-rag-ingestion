import logging 
import os
import time

from io import BytesIO

from openpyxl import load_workbook, Workbook
from tabulate import tabulate

from .base_chunker import BaseChunker

class SpreadsheetChunker(BaseChunker):
    """
    SpreadsheetChunker processes and chunks spreadsheet content, such as Excel files, into manageable pieces for analysis and summarization. 
    It handles both chunking by rows or sheets, allowing users to specify whether to include header rows in each chunk, and ensures that 
    the content size does not exceed a specified token limit.

    The class supports the following operations:
    - Converts spreadsheets into chunkable content.
    - Provides options to chunk either by row or by sheet.
    - Includes optional header rows in chunks.
    - Summarizes large sheets if the content exceeds the maximum chunk size.
    
    Attributes:
    -----------
    max_chunk_size (int): Maximum allowed size of each chunk in tokens.
    chunking_by_row (bool): Whether to chunk by row instead of by sheet.
    include_header_in_chunks (bool): Whether to include header rows in each row-based chunk.
    document_content (str): Processed spreadsheet content ready for chunking.

    Methods:
    --------
    - get_chunks(): Splits the spreadsheet content into manageable chunks, based on the configuration.
    - _spreadsheet_process(): Extracts and processes data from each sheet, including summaries if necessary.
    - _get_sheet_data(sheet): Retrieves data and headers from the given sheet, handling empty cells.
    - _clean_markdown_table(table_str): Cleans up Markdown table strings by removing excessive whitespace.
    """

    def __init__(self, data, max_chunk_size=None, chunking_by_row=None, include_header_in_chunks=None):
        """
        Initializes the SpreadsheetChunker with the provided data and environment configurations.
        
        Args:
            data (str): The spreadsheet content to be chunked.
            max_chunk_size (int, optional): Maximum allowed size of each chunk in tokens. Defaults to an environment variable 'SPREADSHEET_NUM_TOKENS' or 0 if not set.
            chunking_by_row (bool, optional): Whether to chunk by row instead of by sheet. Defaults to an environment variable 'CHUNKING_BY_ROW' or False.
            include_header_in_chunks (bool, optional): Whether to include the header row in each chunk if chunking by row. Defaults to 'INCLUDE_HEADER_IN_CHUNKS' environment variable or False.
        """
        super().__init__(data)
        
        if max_chunk_size is None:
            self.max_chunk_size = int(os.getenv("SPREADSHEET_NUM_TOKENS", 0))
        else:
            self.max_chunk_size = int(max_chunk_size)
        
        if chunking_by_row is None:
            chunking_env = os.getenv("SPREADSHEET_CHUNKING_BY_ROW", "false").lower()
            self.chunking_by_row = chunking_env in ["true", "1", "yes"]
        else:
            self.chunking_by_row = bool(chunking_by_row)
        
        if include_header_in_chunks is None:
            include_header_env = os.getenv("SPREADSHEET_CHUNKING_BY_ROW_INCLUDE_HEADER", "false").lower()
            self.include_header_in_chunks = include_header_env in ["true", "1", "yes"]
        else:
            self.include_header_in_chunks = bool(include_header_in_chunks)

    def get_chunks(self):
        """
        Splits the spreadsheet content into smaller chunks. Depending on the configuration, chunks can be created by sheet or by row.
        - If chunking by sheet, the method summarizes content that exceeds the maximum chunk size.
        - If chunking by row, each row is processed into its own chunk, optionally including the header row.
        
        Returns:
            List[dict]: A list of dictionaries representing the chunks created from the spreadsheet.
        """
        chunks = [] 
        logging.info(f"[spreadsheet_chunker][{self.filename}][get_chunks] Running get_chunks.")
        total_start_time = time.time()

        sheets = self._spreadsheet_process()
        logging.info(f"[spreadsheet_chunker][{self.filename}][get_chunks] Workbook has {len(sheets)} sheets")

        chunk_id = 0
        for sheet in sheets:
            if not self.chunking_by_row:
                # Original behavior: Chunk per sheet
                start_time = time.time()
                chunk_id += 1
                logging.debug(f"[spreadsheet_chunker][{self.filename}][get_chunks][{sheet['name']}] Starting processing chunk {chunk_id} (sheet).")
                table_content = sheet["table"]

                table_content = self._clean_markdown_table(table_content)
                table_tokens = self.token_estimator.estimate_tokens(table_content)
                
                if self.max_chunk_size > 0 and table_tokens > self.max_chunk_size:
                    logging.info(f"[spreadsheet_chunker][{self.filename}][get_chunks][{sheet['name']}] Table has {table_tokens} tokens. Max tokens is {self.max_chunk_size}. Using summary.")
                    table_content = sheet["summary"]

                chunk_dict = self._create_chunk(
                    chunk_id=chunk_id,
                    content=table_content,
                    summary=sheet["summary"] if not self.chunking_by_row else "",
                    embedding_text=sheet["summary"] if (sheet["summary"] and not self.chunking_by_row) else table_content,
                    title=sheet["name"]
                )            
                chunks.append(chunk_dict)
                elapsed_time = time.time() - start_time
                logging.debug(f"[spreadsheet_chunker][{self.filename}][get_chunks][{sheet['name']}] Processed chunk {chunk_id} in {elapsed_time:.2f} seconds.")            
            else:
                # New behavior: Chunk per row
                logging.info(f"[spreadsheet_chunker][{self.filename}][get_chunks][{sheet['name']}] Starting row-wise chunking.")
                headers = sheet.get("headers", [])
                rows = sheet.get("data", [])
                for row_index, row in enumerate(rows, start=1):
                    if not any(cell.strip() for cell in row):
                        continue
                    chunk_id += 1
                    start_time = time.time()
                    logging.debug(f"[spreadsheet_chunker][{self.filename}][get_chunks][{sheet['name']}] Processing chunk {chunk_id} for row {row_index}.")
                    
                    if self.include_header_in_chunks:
                        table = tabulate([headers, row], headers="firstrow", tablefmt="github")
                    else:
                        table = tabulate([row], headers=headers, tablefmt="github")
                    
                    table = self._clean_markdown_table(table)
                    summary = ""
                    
                    table_tokens = self.token_estimator.estimate_tokens(table)
                    if self.max_chunk_size > 0 and table_tokens > self.max_chunk_size:
                        logging.info(f"[spreadsheet_chunker][{self.filename}][get_chunks][{sheet['name']}] Row table has {table_tokens} tokens. Max tokens is {self.max_chunk_size}. Truncating content.")
                        content = table
                        embedding_text = table
                    else:
                        content = table
                        embedding_text = table

                    chunk_dict = self._create_chunk(
                        chunk_id=chunk_id,
                        content=content,
                        summary=summary,
                        embedding_text=embedding_text,
                        title=f"{sheet['name']} - Row {row_index}"
                    )
                    chunks.append(chunk_dict)
                    elapsed_time = time.time() - start_time
                    logging.debug(f"[spreadsheet_chunker][{self.filename}][get_chunks][{sheet['name']}] Processed chunk {chunk_id} in {elapsed_time:.2f} seconds.")
        
        total_elapsed_time = time.time() - total_start_time
        logging.debug(f"[spreadsheet_chunker][{self.filename}][get_chunks] Finished get_chunks. Created {len(chunks)} chunks in {total_elapsed_time:.2f} seconds.")

        return chunks

    def _spreadsheet_process(self):
        """
        Extracts and processes each sheet from the spreadsheet, converting the content into Markdown table format.

        If a sheet has more rows than SPREADSHEET_MAX_ROWS_PER_SPLIT (default 200), split the sheet into partitions
        and upload each partition as a separate XLSX file back to the same container/folder as the original file.

        After successful upload of all partitions, delete the original blob. Stop processing (return empty list)
        so the newly created partitions can be processed on the next run.

        Returns:
            List[dict]: A list of dictionaries, where each dictionary contains sheet metadata. If a split was performed,
            an empty list is returned to stop further processing in the current run.
        """
        logging.debug(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process] Starting blob download.")
        blob_data = self.document_bytes
        blob_stream = BytesIO(blob_data)
        logging.debug(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process] Starting openpyxl load_workbook.")
        workbook = load_workbook(blob_stream, data_only=True)

        sheets = []
        total_start_time = time.time()

        # Determine max rows per partition (default 200). Can be overridden via env var SPREADSHEET_MAX_ROWS_PER_SPLIT
        max_rows_env = os.getenv("SPREADSHEET_MAX_ROWS_PER_SPLIT", "")
        try:
            max_rows = int(max_rows_env) if max_rows_env else 200
        except ValueError:
            logging.warning(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process] Invalid SPREADSHEET_MAX_ROWS_PER_SPLIT='{max_rows_env}', falling back to 200.")
            max_rows = 200

        # Prepare to upload partitions back to source (same container/folder as original)
        store_on_source_env = os.getenv("SPREADSHEET_STORE_PARTITIONS_ON_SOURCE", "true").lower()
        store_on_source = store_on_source_env in ["true", "1", "yes"]

        container_client = None
        original_blob_relpath = None
        base_blob_dir = ""
        storage_account_base_url = None
        container_name = None

        if store_on_source:
            try:
                # Parse original file URL to get account & container & relative path
                from urllib.parse import urlparse
                from tools.blob import BlobContainerClient

                parsed = urlparse(self.file_url)
                storage_account_base_url = f"{parsed.scheme}://{parsed.netloc}"
                path_parts = parsed.path.split('/')
                if len(path_parts) >= 2:
                    container_name = path_parts[1]
                    original_blob_relpath = '/'.join(path_parts[2:])
                    base_blob_dir = os.path.dirname(original_blob_relpath)
                else:
                    logging.warning(f"[spreadsheet_chunker][{self.filename}] Unexpected blob URL path format: {parsed.path}")
                    store_on_source = False

                if store_on_source:
                    container_client = BlobContainerClient(storage_account_base_url, container_name)
                    logging.debug(f"[spreadsheet_chunker][{self.filename}] Initialized BlobContainerClient for container '{container_name}'.")
            except Exception as e:
                logging.warning(f"[spreadsheet_chunker][{self.filename}] Could not initialize BlobContainerClient: {e}")
                container_client = None
                store_on_source = False

        # directory fallback for local partition files (if upload fails or not configured)
        partition_output_dir = os.getenv("SPREADSHEET_PARTITION_OUTPUT_DIR", "./tmp/spreadsheet_partitions")
        try:
            os.makedirs(partition_output_dir, exist_ok=True)
        except Exception:
            pass

        splits_performed = False
        all_uploads_succeeded = True

        # Iterate sheets and split any sheet with more than max_rows data rows
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            data, headers = self._get_sheet_data(sheet)

            if len(data) > max_rows:
                # need to split this sheet
                logging.info(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Splitting sheet with {len(data)} rows into parts of {max_rows} rows.")
                splits_performed = True
                total_rows = len(data)
                num_parts = (total_rows + max_rows - 1) // max_rows

                for part_index in range(num_parts):
                    start_idx = part_index * max_rows
                    end_idx = min(start_idx + max_rows, total_rows)
                    part_rows = data[start_idx:end_idx]

                    # Build a safe partition filename
                    import re
                    safe_sheet_name = re.sub(r"[^A-Za-z0-9_\-]", "_", sheet_name)[:31]
                    original_name_no_ext = os.path.splitext(self.filename)[0]
                    partition_filename = f"{original_name_no_ext}_{safe_sheet_name}_part_{part_index+1}.xlsx"

                    # If we have a container and base dir, upload to same folder; else write locally
                    if store_on_source and container_client is not None and base_blob_dir is not None:
                        blob_name = f"{base_blob_dir}/{partition_filename}" if base_blob_dir else partition_filename
                    else:
                        blob_name = partition_filename

                    # Create temporary workbook for the partition
                    import tempfile
                    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    temp_path = temp.name
                    temp.close()

                    try:
                        wb = Workbook()
                        ws = wb.active
                        ws.title = safe_sheet_name if safe_sheet_name else "Sheet1"

                        # Write headers and rows
                        if headers:
                            ws.append([str(h) for h in headers])
                        for row in part_rows:
                            ws.append([str(c) for c in row])

                        wb.save(temp_path)

                        if store_on_source and container_client is not None:
                            try:
                                # Upload and set sensitivity metadata to 'general'
                                try:
                                    blob_client_under = container_client.container_client.get_blob_client(blob_name)
                                    with open(temp_path, 'rb') as data:
                                        blob_client_under.upload_blob(data, overwrite=True, metadata={"sensitivity": "general"})
                                    logging.info(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Uploaded partition '{blob_name}' to container '{container_name}' with sensitivity='general'.")
                                except Exception as upload_err_inner:
                                    # Fall back to previous path of attempting a wrapper upload (if any) or log and fallback to local
                                    raise upload_err_inner
                                
                            except Exception as upload_err:
                                 all_uploads_succeeded = False
                                 logging.warning(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Failed to upload partition '{blob_name}': {upload_err}")
                                 # fallback: write locally
                                 try:
                                     local_path = os.path.join(partition_output_dir, partition_filename)
                                     os.replace(temp_path, local_path)
                                     logging.info(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Wrote partition locally to '{local_path}' as fallback.")
                                 except Exception as local_err:
                                     logging.warning(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Failed to write partition locally: {local_err}")
                                 finally:
                                     # Remove temp if present
                                     try:
                                         if os.path.exists(temp_path):
                                             os.remove(temp_path)
                                     except Exception:
                                         pass
                        else:
                            # Not uploading; write partition locally
                            try:
                                local_path = os.path.join(partition_output_dir, partition_filename)
                                os.replace(temp_path, local_path)
                                logging.info(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Wrote partition locally to '{local_path}'.")
                            except Exception as local_err:
                                logging.warning(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Failed to write partition locally: {local_err}")
                                try:
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                except Exception:
                                    pass

                    except Exception as e:
                        all_uploads_succeeded = False
                        logging.warning(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Error creating partition file: {e}")
                        # Ensure temp removed
                        try:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        except Exception:
                            pass

                # finished splitting this sheet; continue to next sheet

        # If any splitting was performed, delete original blob only if uploads succeeded
        if splits_performed:
            if store_on_source and container_client is not None and all_uploads_succeeded and original_blob_relpath:
                try:
                    container_client.delete_blob(original_blob_relpath)
                    logging.info(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process] Deleted original blob '{original_blob_relpath}' after creating partitions.")
                except Exception as e:
                    logging.warning(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process] Failed to delete original blob '{original_blob_relpath}': {e}")
            # Stop processing now; partitions were created and uploaded/written to disk. Next run should pick them up.
            logging.info(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process] Splitting complete; stopping further processing so partitions can be processed in subsequent runs.")
            return []

        # No splitting required; proceed with original behavior
        for sheet_name in workbook.sheetnames:
            logging.info(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Started processing.")
            start_time = time.time()
            sheet_dict = {}            
            sheet = workbook[sheet_name]
            data, headers = self._get_sheet_data(sheet)
            sheet_dict["headers"] = headers
            sheet_dict["data"] = data
            table = tabulate(data, headers=headers, tablefmt="grid")
            table = self._clean_markdown_table(table)
            sheet_dict["table"] = table

            if not self.chunking_by_row:
                prompt = f"Summarize the table with data in it, by understanding the information clearly.\n table_data:{table}"
                try:
                    summary = self.aoai_client.get_completion(prompt, max_tokens=2048)
                except Exception:
                    summary = ""
                sheet_dict["summary"] = summary
                logging.debug(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Generated summary.")
            else:
                sheet_dict["summary"] = ""
                logging.debug(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Skipped summary generation (chunking by row).")

            elapsed_time = time.time() - start_time
            logging.debug(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process][{sheet_name}] Processed in {elapsed_time:.2f} seconds.")
            sheet_dict['name'] = sheet_name
            sheets.append(sheet_dict)

        total_elapsed_time = time.time() - total_start_time
        logging.debug(f"[spreadsheet_chunker][{self.filename}][spreadsheet_process] Total processing time: {total_elapsed_time:.2f} seconds.")

        return sheets

    def _get_sheet_data(self, sheet):
        """
        Retrieves data and headers from the given sheet. Each row's data is processed into a list format, ensuring that empty rows are excluded.

        Args:
            sheet (Worksheet): The worksheet object to extract data from.

        Returns:
            Tuple[List[List[str]], List[str]]: A tuple containing a list of row data and a list of headers.
        """
        data = []
        for row in sheet.iter_rows(min_row=2):  # Start from the second row to skip headers
            row_data = []
            for cell in row:
                cell_value = cell.value
                if cell_value is None:
                    cell_value = ""
                cell_text = str(cell_value)
                row_data.append(cell_text)
            if "".join(row_data).strip() != "":
                data.append(row_data)

        headers = [cell.value if cell.value is not None else "" for cell in sheet[1]]
        return data, headers
    
    def _clean_markdown_table(self, table_str):
        """
        Cleans up a Markdown table string by removing excessive whitespace from each cell.

        Args:
            table_str (str): The Markdown table string to be cleaned.

        Returns:
            str: The cleaned Markdown table string with reduced whitespace.
        """
        cleaned_lines = []
        lines = table_str.splitlines()

        for line in lines:
            if set(line.strip()) <= set('-| '):
                cleaned_lines.append(line)
                continue

            cells = line.split('|')
            stripped_cells = [cell.strip() for cell in cells[1:-1]]
            cleaned_line = '| ' + ' | '.join(stripped_cells) + ' |'
            cleaned_lines.append(cleaned_line)

        return '\n'.join(cleaned_lines)