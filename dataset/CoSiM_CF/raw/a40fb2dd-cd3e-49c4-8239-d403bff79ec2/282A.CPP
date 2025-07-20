// my first program in C++
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  int nr = 0;
  for (auto i = 0; i < n; i++) {
    string line;
    cin >> line;
    if (line.find('+') != string::npos) {
      nr++;
    } else {
      nr--;
    }
  }
  cout << nr;
  return 0;
}