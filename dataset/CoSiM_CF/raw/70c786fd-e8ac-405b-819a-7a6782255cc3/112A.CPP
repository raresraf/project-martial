#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string str;
  string str2;

  transform(str.begin(), str.end(), str.begin(), ::tolower);
  transform(str2.begin(), str2.end(), str2.begin(), ::tolower);
  if (str < str2) {
    cout << -1;
  } else if (str > str2) {
    cout << 1;
  } else {
    cout << 0;
  }
  return 0;
}
