#include "string"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  string s, str = "hello";
  cin >> s;
  int pos = 0, count = 0;

  for (int i = 0; i < s.length(); i++) {
    if (s[i] == str[pos]) {
      count++;
      pos++;
    }
  }

  if (count == 5)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
