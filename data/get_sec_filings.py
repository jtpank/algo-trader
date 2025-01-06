import requests
import json
import xmltodict
import csv
import os
# import requests_random_user_agent


#Format: https://www.sec.gov/Archives/edgar/data/
# 1690820/
# 000169082024000390/
# 0001690820-24-000390-index.html

def gather_cik_data(url: str):
    headers = {
        'User-Agent': 'Sample Company Name AdminContact@samplecompanydomain.com',
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        try:
            json_data = response.json()
            return json_data
        except ValueError:
            print("Error: Response is not in valid JSON format")
    else:
        print(f"Error: {response.status_code}")


def get_xml_to_json_data(url: str):# Define custom User-Agent header to avoid blocking
    headers = {
        'User-Agent': 'Sample Company Name AdminContact@samplecompanydomain.com',
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        try:
            xml_dict = xmltodict.parse(response.text)
            json_data = json.dumps(xml_dict, indent=4)
            return json_data
        except ValueError:
            print("Error: Response is not in valid JSON format")
    else:
        print(f"Error: {response.status_code}")

def save_to_json_files(form_4_xml_array):
    for i in form_4_xml_array:
        print(i)
        xml_json_ownership = get_xml_to_json_data(i)
        json_data = json.loads(xml_json_ownership)
        save_dir = '/Users/justin/algo/algo-trader/data/cvna-sec-filings/ownership-json'
        # filename = (i.split("/")[-1]).split(".")[0] + ".json"
        filename = (i.split("/")[-2]) + "-ownership" + ".json"
        save_file = os.path.join(save_dir, filename)
        with open(save_file, "w") as f:
            json.dump(json_data, f, indent=4)

def get_cik_json_data():
    ticker = "CVNA"
    # base_url = "https://www.sec.gov/Archives/edgar/data/"
    # cik = "1690820"
    # section_key = "000169082024000390"
    # index_format = "0001690820-24-000390-index.html"

    # full_url = base_url + key + section_key + index_format
    # print(full_url)
    

    new_url = "https://www.sec.gov/Archives/edgar/data/1690820/000169082024000393/wk-form4_1734128836.xml"

    url = "https://data.sec.gov/submissions/CIK0001690820.json"
    cik_json_data = gather_cik_data(url)
    form_url_dict = {}
    base_url = "https://www.sec.gov/Archives/edgar/data/1690820/"
    for key, value in cik_json_data.items():
        if key == 'filings':
            for filings_key, filings_value in value.items():
                #recent or files
                if filings_key == 'recent':
                    for recent_filings_key, recent_filings_value in filings_value.items():
                        form_type = filings_value['form']
                        access_numbers = filings_value['accessionNumber']
                        primary_documents = filings_value['primaryDocument']
                        assert len(form_type) == len(access_numbers)
                        assert len(access_numbers) == len(primary_documents)
                        for idx, it in enumerate(form_type):
                            if primary_documents[idx].split(".")[-1] != "xml" or "doc4" in primary_documents[idx]:
                                continue
                            trailing_url_string = str(access_numbers[idx]) + "/" + str(primary_documents[idx])
                            trailing_arr = trailing_url_string.split("/")
                            trailing_arr_front = ''.join(trailing_arr[0].split("-"))
                            trailing_arr_end = trailing_arr[-1]
                            full_path_string = base_url + trailing_arr_front + "/" + trailing_arr_end
                            if it in form_url_dict.keys():
                                form_url_dict[it].append(full_path_string)
                            else:
                                form_url_dict[it] = [full_path_string]
    form_4_array = []
    for k, v in form_url_dict.items():
        if k == '4':
            form_4_array = v
    # print(form_4_array)
    # print(len(set(form_4_array)))
    form_4_set_sorted = sorted(set(form_4_array))
    ownership_array = [i for i in form_4_set_sorted if "ownership" in i]
    form_4_xml_array = [i for i in form_4_set_sorted if "wk-form4" in i]
    save_to_json_files(ownership_array)


def parse_json_data():
    # print(json.dumps(json_data, indent=4))
    json_dir = '/Users/justin/algo/algo-trader/data/cvna-sec-filings/form-4-json'
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, 'r') as json_file:
                json_data = json.load(json_file)
                print(json_data['ownershipDocument']['reportingOwner']['reportingOwnerId']['rptOwnerName'])
                non_derivative_table = json_data['ownershipDocument']['nonDerivativeTable']
                non_derivative_transaction_array = non_derivative_table['nonDerivativeTransaction']
                if len(non_derivative_table) > 0:
                    for i in non_derivative_transaction_array:
                        print(i)
                # print(non_derivative_table['nonDerivativeTransaction'][0].keys())
                # print(non_derivative_table['nonDerivativeHolding'][0].keys())
            break

def main():
    parse_json_data()


if __name__=="__main__":
    main()