#include <bits/stdc++.h>
using namespace std;
int F(string s) {
  long long int i = 0, test = 0;
  while (s[i] != '\0') {
    if (s[i] != '4' && s[i] != '7')
      test++;
    i++;
  }
  return test;
}
int f(string s) {
  long long int i = 0, n = 0, test;
  while (s[i] != '\0') {
    if (s[i] == '4' || s[i] == '7')
      n++;
    i++;
  }
  ostringstream s1;
  string s2;
  s1 << n;
  s2 = s1.str();
  test = F(s2);
  return test;
}
int main() {
  long long int n, len, test;
  cin >> n;
  ostringstream s1;
  string s2;
  s1 << n;
  s2 = s1.str();
  test = f(s2);
  if (test)
    cout << "NO";
  else
    cout << "YES";
  return 0;
}
