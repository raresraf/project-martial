// http://codeforces.com/problemset/problem/118/A

#include <iostream>
using namespace std;

int main() {
  string s;

  getline(cin, s);
  int sz = s.length();
  for (int i = 0; i < sz; i++)
    if (s[i] <= 'Z')
      s[i] += 32;
  for (int i = 0; i < sz;)
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'i' ||
        s[i] == 'e' || s[i] == 'u')
      s.erase(i, 1);
    else
      i++;
  for (int i = 0; i < s.length(); i++)
    cout << "." << s[i];
  return 0;
}
