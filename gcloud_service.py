from google.cloud import storage

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name, credentials_file):
    storage_client = storage.Client.from_service_account_json(credentials_file)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)

    print("Uploaded to {}/{}".format(bucket_name, destination_blob_name))


def main():
    target_folder = "ego4d"
    # file_name = "ego4d_clip_visual_features.pt"
    # file_name = "ego4d_clip_textual_features.pt"
    # file_name = "ego4d_clip_textual_features.pt"
    # file_name = "ego4d_error_list.txt"
    # file_name = "ego4d_more_error_list.txt"
    file_name = "ego4d_query_list.jsonl"
    BUCKET_NAME = "image_generation_for_snag"
    SOURCE_FILE_PATH = "/home/nguyenpk/image_generation/{}".format(file_name)
    DESTINATION_BLOB_NAME = target_folder + '/' + file_name
    CREDENTIALS_FILE = "credentials_file.json"

    upload_to_gcs(BUCKET_NAME, SOURCE_FILE_PATH, DESTINATION_BLOB_NAME, CREDENTIALS_FILE)


if __name__ == '__main__':
    main()