#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  cin >> str;
  string str1 = "0000000";
  string str2 = "1111111";
  std::size_t found0 = str.find(str1);
  std::size_t found1 = str.find(str2);
  if ((found0 != std::string::npos) || (found1 != std::string::npos)) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}