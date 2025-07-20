#include <cstdio>
#include <iostream>
#include <string.h>

using namespace std;

string convert_lower(string ch) {
  for (int i = 0; i < ch.size(); i++) {
    ch[i] = tolower(ch[i]);
  }

  return ch;
}

int main() {
  string first, second;
  cin >> first >> second;
  first = convert_lower(first);
  second = convert_lower(second);
  int res = 0;
  for (int i = 0; i < first.size(); i++) {
    if (first[i] == second[i]) {

      continue;
    } else if (first[i] > second[i]) {
      res = 1;
      break;
    } else {
      res = -1;
      break;
    }
  }
  cout << res;
  return 0;
}
