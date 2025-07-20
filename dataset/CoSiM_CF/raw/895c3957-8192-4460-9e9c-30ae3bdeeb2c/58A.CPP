#include <bits/stdc++.h>

using namespace std;
char *init_hello() {
  char tmp[6] = "hello";
  return tmp;
}
char *init_chatroot_io() {

  char *s = (char *)malloc(sizeof(char) * 100);
  cin >> s;
  return s;
}
int main() {
  char t[] = "hello";
  char *hello = t;
  char *s = (char *)malloc(sizeof(char) * 100);
  cin >> s;
  int count = 0;
  while (*s) {
    if (*s == *hello) {
      *hello++;
      count++;
      *s++;
    } else
      *s++;
  }
  if (count == 5)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}