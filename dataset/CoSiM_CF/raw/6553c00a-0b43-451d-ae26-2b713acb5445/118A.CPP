// my first program in C++
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  string data;
  cin >> data;
  transform(data.begin(), data.end(), data.begin(), ::tolower);
  for (auto c : data) {
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y') {

    } else {
      cout << "." << c;
    }
  }
  return 0;
}