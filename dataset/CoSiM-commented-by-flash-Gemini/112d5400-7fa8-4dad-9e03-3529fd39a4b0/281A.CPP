#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  string line;
  cin >> line;
  line[0] = toupper(line[0]);
  cout << line;
  return 0;
}