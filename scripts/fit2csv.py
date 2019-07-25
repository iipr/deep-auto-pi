###############################################################################
#                                                                             #
#           This script extracts data from all .fit files in a folder         #
#               and saves them as .csv files in the same folder.              #
#                              July 2019, iipr                                #
#                                                                             #
#                                 Reference:                                  #
#  https://maxcandocia.com/article/2017/Sep/22/converting-garmin-fit-to-csv/  #
#                                                                             #
###############################################################################

import csv, os, fitparse, pytz

allowed_fields = ['utc_timestamp', 'timestamp_ms', 
                  'position_lat','position_long', 'enhanced_altitude', 
                  'enhanced_speed', 'velocity']
required_fields = ['utc_timestamp', 'timestamp_ms', 
                   'position_lat', 'position_long']

UTC = pytz.UTC
CST = pytz.timezone('US/Central')

def write_fitfile_to_csv(fitfile, output_file='test_output.csv'):
    messages = fitfile.messages
    data = []
    #s = set()
    for m in messages:
        skip=False
        if not hasattr(m, 'fields'):
            continue
        fields = m.fields
        # Check for important data types
        mdata = {}
        for field in fields:
            #s.add(field.name)
            if field.name in allowed_fields:
                if field.name == 'utc_timestamp':
                    mdata[field.name] = UTC.localize(field.value)#.astimezone(CST)
                else:
                    mdata[field.name] = field.value
        for rf in required_fields:
            if rf not in mdata:
                skip = True
        if not skip:
            data.append(mdata)
    # Write to csv
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(allowed_fields)
        for entry in data:
            writer.writerow([str(entry.get(k, '')) for k in allowed_fields])
    print('Wrote {}'.format(output_file))
    #print('Set:', s)
    
def parse_fitfiles(path='.', outpath='.'):
    files = os.listdir(path)
    fit_files = [file for file in files if file[-4:].lower() == '.fit']
    for file in fit_files:
        new_filename = outpath + file[:-4] + '.csv'
        if os.path.exists(new_filename):
            #print('{} already exists... skipping...'.format(new_filename))
            continue
        fitfile = fitparse.FitFile(path + file,
                                   data_processor=fitparse.StandardUnitsDataProcessor())
        print('Converting {}...'.format(file))
        write_fitfile_to_csv(fitfile, new_filename)
    print('Finished conversions!')