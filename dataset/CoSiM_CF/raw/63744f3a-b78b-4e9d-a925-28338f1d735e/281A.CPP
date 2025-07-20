#include <iostream>
using namespace std;
int main() {
  string s, g;
  cin >> s;
  if (s[0] > 90)
    s[0] = s[0] - 'a' + 'A';
  cout << s << endl;
}