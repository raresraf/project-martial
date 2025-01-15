#include <iostream>
using namespace std;
int main() {
  string s, s1 = "hello";
  cin >> s;
  int j = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == s1[j])
      j++;
  }
  if (j == 5)
    cout << "YES\n";
  else
    cout << "NO\n";
}
