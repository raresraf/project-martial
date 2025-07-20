#include <algorithm>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string string1, string2;
  cin >> string1 >> string2;
  transform(string1.begin(), string1.end(), string1.begin(), ::toupper);
  transform(string2.begin(), string2.end(), string2.begin(), ::toupper);
  int iter(0), length(string1.length()), string1value(0), string2value(0);
  for (iter = 0; iter < length; iter++) {
    string1value += int(string1[iter]);
    string2value += int(string2[iter]);
  }
  if (string1.compare(string2) < 0) {
    cout << "-1";
  } else if (string1.compare(string2) == 0) {
    cout << "0";
  } else {
    cout << "1";
  }
  return 0;
}