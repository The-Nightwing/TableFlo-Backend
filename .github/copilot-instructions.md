# GitHub Copilot Instructions

This file guides GitHub Copilot and human developers to produce consistent, secure, and maintainable code for this service.  
This project is an API-based data processing platform for **Authentication**, **Password Management**, **File Uploading**, **File Processing**, **Dataframe Operations**, and **Process Management**.

---

## ✅ GENERAL RULES

- **Base URL:** `http://tableflow1.ap-south-1.elasticbeanstalk.com/`
- For any authenticated API, always add this required header:
X-User-Email: user@example.com

- Use **JWT tokens** where needed for session security.
- Use **HTTP status codes properly**:
- `200` → OK
- `201` → Created
- `400` → Bad Request
- `401` → Unauthorized
- `404` → Not Found
- `500` → Internal Server Error

---

## ✏️ CODE STYLE GUIDELINES

- API routes use **snake_case**: `/api/validate-otp/`
- JSON keys use **camelCase** or **snake_case** consistently.
- Internal variables follow **camelCase**.
- Use clear, descriptive variable names: `processId`, `tableName`, `operationId`.

---

## 🔐 AUTHENTICATION MODULE

**Endpoints:**
- **Register**
- `POST /api/register/`
- Body: `{ "name": "John Doe", "email": "...", "company_name": "...", "password": "..." }`
- 201 Created → `{"message": "OTP sent to email"}`

- **Validate Registration OTP**
- `POST /api/validate-otp/`
- Body: `{ "email": "...", "otp": "..." }`

- **Login**
- `POST /api/login/`
- Body: `{ "email": "...", "password": "..." }`
- 200 OK → `{ "message": "...", "token": "...", "email": "..." }`

---

## 🔑 PASSWORD MANAGEMENT MODULE

**Endpoints:**
- **Forgot Password**
- `POST /api/forgot-password/` → `{ "email": "..." }`
- **Validate OTP**
- `POST /api/validate-forgot-password-otp/`
- **Reset Password**
- `POST /api/reset-password/`
- Body: `{ "email": "...", "otp": "...", "newPassword": "..." }`

---

## 📂 FILE MANAGEMENT

**Endpoints:**
- **Process Uploaded Files**
- `POST /api/process-uploaded-files/`
- Body: `{ "uploadedFiles": { "file1": {...} }, "email": "..." }`

- **List User Files**
- `GET /api/list-files/`
- Always include `X-User-Email`.

- **Delete File**
- `POST /api/list-files/<fileID>/delete`

- **Get File Details**
- `GET /api/get-file-details?fileId=...`

---

## 📊 DATA PROCESSING

**Store Dataframe:**
- `POST /api/store-dataframe-for-process/`
- `{ "processId": "...", "tableName": "...", "fileId": "...", "sheetName": "..." }`

**Get Table Details:**
- `GET /api/get-table-details?processId=...&tableName=...`

**Get Table Data:**
- `GET /api/get-table-data?processId=...&tableName=...&page=1&perPage=100`

---

## 🧩 DATAFRAME OPERATIONS

**Column Processing:**
- `POST /api/edit-file/edit/`
- Body: `{ "processId": "...", "tables": [{ "tableName": "...", "columnSelections": {...}, "columnTypes": {...}, "datetimeFormats": {...} }] }`

**Add Column:**
- `POST /api/add-column/apply/`
- Supports `pattern`, `calculate`, `concatenate`, `conditional`.

**Merge Files:**
- `POST /api/merge-files/merge`
- `{ "processId": "...", "table1Name": "...", "table2Name": "...", "mergeType": "horizontal|vertical" }`

**Group Pivot:**
- `POST /api/group-pivot/generate`
- `{ "processId": "...", "tableName": "...", "rowIndex": ["..."], "pivotValues": [{...}] }`

**Sort & Filter:**
- `POST /api/sort-filter/apply`
- `{ "processId": "...", "tableName": "...", "outputTableName": "...", "sortConfig": [...], "filterConfig": [...] }`

**Replace / Rename / Reorder:**
- `POST /api/operations/process/apply`

**Reconcile:**
- `POST /api/merge-files/process/reconcile`
- `{ "processId": "...", "sourceTableNames": ["Table 1", "Table 2"], ... }`

---

## ⚙️ PROCESS MANAGEMENT

**Start Process:**
- `POST /api/process/start`
- `{ "name": "Process Name" }`

**Update Process:**
- `PATCH /api/process/<processId>`
- `{ "name": "New Name" }`

**Get Process Dataframes:**
- `GET /api/process/<processId>/dataframes`

**Add Operations to Process:**
- `POST /api/process/<processId>/operations`
- `{ "dataframeOperationId": "...", "operationType": "...", "sequence": 1.0 }`

**Get All Process Operations:**
- `GET /api/process/<processId>/operations`

---

## 🛡️ SECURITY BEST PRACTICES

- Never hardcode secrets or tokens.
- Always validate input — especially emails, OTPs, file IDs.
- Handle all edge cases: missing fields, invalid formats, unauthorized users.
- Log unexpected errors for debugging but do not expose stack traces in responses.

---

## ✔️ COPILOT BEHAVIOR TIPS

- Prefer type-safe request parsing.
- Generate clear API doc strings.
- Always wrap DB/storage calls with error handling.
- Reuse shared utilities for input validation and JWT decoding.
- Suggest sample unit tests for each endpoint when writing handler code.

---

## 📣 QUESTIONS?

For design doubts or new API additions, check this file and align your implementation accordingly.  
When unsure, default to **clean, secure, RESTful practices**.

---