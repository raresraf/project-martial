#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;

  transform(s.begin(), s.begin() + 1, s.begin(), ::toupper);

  cout << s << endl;
  return 0;
}
