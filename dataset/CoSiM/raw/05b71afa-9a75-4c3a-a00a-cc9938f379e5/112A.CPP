#include <iostream>
#include <string>
using namespace std;

void toLowerCase(string &str) {
  for (int i = 0; i < str.size(); i++)
    str[i] = tolower(str[i]);
}

int main() {
  string first, second;
  int i, length, out;
  cin >> first >> second;
  length = first.size();
  toLowerCase(first);
  toLowerCase(second);
  for (i = 0; i < length; i++) {
    if (first[i] != second[i]) {
      if (first[i] < second[i]) {
        out = -1;
        break;
      } else {
        out = 1;
        break;
      }
    }
    out = 0;
  }
  cout << out << endl;
  return 0;
}
