# OneDrive Upload

`scripts/upload_folder_to_onedrive.py` uploads a local folder recursively using
Microsoft Graph device-code login and resumable upload sessions.

## Microsoft App Setup

Create an app registration in Microsoft Entra / Azure:

1. Go to <https://portal.azure.com/>
2. Search for **App registrations**
3. Create a new registration
4. Supported account types: personal Microsoft accounts if you use Outlook/OneDrive personal
5. Copy the **Application (client) ID**
6. Under **Authentication**, enable public client/native flows if needed
7. Under **API permissions**, add delegated Microsoft Graph permission:
   `Files.ReadWrite.All`

## Upload

On Lightning:

```bash
pip install -r requirements-lightning.txt

python scripts/upload_folder_to_onedrive.py \
  --source data \
  --remote ImageCLEF_Backup/data \
  --client-id YOUR_CLIENT_ID
```

The script will print a device login URL and code. Open it in your browser,
sign into OneDrive, and approve access.

For a safer first check:

```bash
python scripts/upload_folder_to_onedrive.py \
  --source data \
  --remote ImageCLEF_Backup/data \
  --client-id YOUR_CLIENT_ID \
  --dry-run
```

To skip files already in OneDrive:

```bash
python scripts/upload_folder_to_onedrive.py \
  --source data \
  --remote ImageCLEF_Backup/data \
  --client-id YOUR_CLIENT_ID \
  --skip-existing
```

