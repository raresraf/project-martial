#include <cctype>
#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;

  s.at(0) = toupper(s.at(0));

  cout << s << endl;

  return 0;
}
