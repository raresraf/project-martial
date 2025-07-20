
#include <iostream>

#include <string>

using namespace std;

int main() {
  // your code goes here
  string str;
  cin >> str;
  string str1 = "1111111";
  string str2 = "0000000";
  size_t found = str.find(str1);
  if (found != string::npos) {
    cout << "YES" << endl;
    return 0;
  }
  for (int i = 0; i < 7; i++) {
    str1[i] = str2[i];
  }

  size_t found1 = str.find(str1);
  if (found1 != string::npos) {
    cout << "YES" << endl;
    return 0;
  }
  cout << "NO" << endl;

  return 0;
}