#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  string line;
  cin >> line;
  string temp = "hello";
  int index = 0;
  for (auto i = 0; i < line.length() && index < temp.length(); i++) {
    if (line[i] == temp[index]) {
      index++;
    }
  }
  if (index == temp.length())
    cout << "YES";
  else
    cout << "NO";
  return 0;
}