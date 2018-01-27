#!/usr/bin/env python
import os
import sys
import time

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import datetime

class GDriveHandler():
    def __init__(self):
        self.drive = None
        self.authenticate()

    def authenticate(self):
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile("credentials.txt")
        if gauth.credentials is None:
            gauth.CommandLineAuth()
        elif gauth.access_token_expired:
            #gauth.CommandLineAuth()
            gauth.Refresh()
        else:
            gauth.Authorize()

        gauth.SaveCredentialsFile("credentials.txt")

        self.drive = GoogleDrive(gauth)
        self.parent_folder_id = '1nDg3jbOW2Uo27DBPOLaOQPjajqalGU6m'

    def upload_file(self, filename):
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(self.parent_folder_id)}).GetList()
        folder_map = {}
        for file1 in file_list:
            folder_map[file1['title']] = file1['id']

        today = str(datetime.datetime.now().date())

        folder_id = ''
        if today in folder_map:
            folder_id = folder_map[today]
        else:
            print('Creating Folder.')
            folder_metadata = {
                'title' : today,
                'mimeType' : 'application/vnd.google-apps.folder',
                "parents": [{"kind": "drive#fileLink", "id": self.parent_folder_id}]
            }
            folder = self.drive.CreateFile(folder_metadata)
            folder.Upload()

            folder_title = folder['title']
            folder_id = folder['id']
            print('title: %s, id: %s' % (folder_title, folder_id))

        # # Upload file to folder.
        if folder_id != '':
            filetitle = filename.split("/")[-1]
            f = self.drive.CreateFile({'title': filetitle, "parents": [{"kind": "drive#fileLink", "id": folder_id}]})
            # Make sure to add the path to the file to upload below.
            f.SetContentFile(filename)
            f.Upload()
            print('Upload done.')

    def _get_recent_files(self):
        """Get recent dates from today to num_days."""
        num_days = 7
        file_list = []
        for i in range(num_days):
            x = datetime.datetime.now() - datetime.timedelta(days=i)
            file_list.append(str(x.date()))

        return file_list

    def delete_old_files(self):
        """Delete files older than a certain date limit."""
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(self.parent_folder_id)}).GetList()
        file_ids = []
        recent_files = self._get_recent_files()

        for file1 in file_list:
            if file1['title'] not in recent_files:
                file_ids.append(file1['id'])

        self.delete_files(file_ids)

    def delete_files(self, file_ids):
        """Delete a list of files based on file_ids."""
        for fid in file_ids:
            f = self.drive.CreateFile({'id': fid})
            f.Delete()

if __name__ == '__main__':
    obj = GDriveHandler()
    obj.delete_old_files()
