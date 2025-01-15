#include <iostream>
#include <regex>

using namespace std;

int main() {
  string s;
  cin >> s;
  if (regex_match(s, regex("[0-1]*0000000[0-1]*")) ||
      regex_match(s, regex("[0-1]*1111111[0-1]*")))
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
