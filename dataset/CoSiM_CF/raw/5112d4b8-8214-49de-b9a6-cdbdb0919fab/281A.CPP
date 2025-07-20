#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int main() {
  int hasil;
  string a;
  cin >> a;
  if (a[0] >= 97 && a[0] <= 122)
    a[0] -= 32;
  for (int i = 0; i < a.length(); i++)
    printf("%c", a[i]);
}
