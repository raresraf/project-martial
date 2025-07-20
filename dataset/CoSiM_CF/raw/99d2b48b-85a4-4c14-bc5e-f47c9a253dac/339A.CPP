/// Problem Name: Helpful Maths
/// Problem Link: https://codeforces.com/problemset/problem/339/A

#include <cstdio>
#include <iostream>
using namespace std;
string s1, s2;
void b_sort(string s) {
  int l = s.size();
  for (int i = 0; i < l; i++)
    for (int j = 0; j < l - i - 1; j++)
      if (s[j] > s[j + 1]) {
        char ch = s[j];
        s[j] = s[j + 1];
        s[j + 1] = ch;
      }
  l = s.size();
  for (int i = 0; i < l; i++) {
    printf("%c", s[i]);
    if (i != l - 1)
      printf("+");
  }
}
int main() {
  cin >> s1;
  int l = s1.size();
  for (int i = 0; i < l; i++)
    if (s1[i] != '+')
      s2 += s1[i];
  b_sort(s2);
  return puts(""), 0;
}
