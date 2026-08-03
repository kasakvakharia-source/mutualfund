from mftool import Mftool

mf = Mftool()
test_code = "139201"

details = mf.get_scheme_details(test_code)
print(details)