// my first program in C++
#include <algorithm>
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
  string string1, string2;
  cin >> string1 >> string2;

  transform(string1.begin(), string1.end(), string1.begin(), ::tolower);
  transform(string2.begin(), string2.end(), string2.begin(), ::tolower);
  int res = 0;
  for (size_t i = 0; i < string1.length(); i++) {
    if (int(string1[i]) > int(string2[i])) {
      res = 1;
      break;
    }
    if (int(string1[i]) < int(string2[i])) {
      res = -1;
      break;
    }
  }
  cout << res;

  return 0;
}