#include <iostream>
#include <string>
using namespace std;
int main() {
  int n, m = 0;
  string s;
  cin >> n;
  cin >> s;
  for (int i = 0; i < s.length(); i++)
    m = s[i] == s[i + 1] ? m + 1 : m;
  cout << m << endl;
  return 0;
}
