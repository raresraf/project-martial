#include <ctype.h>
#include <iostream>
#include <string.h>
using namespace std;
int main() {
  int t = 0;
  string a;
  cin >> a;
  char c;

  while (a[t]) {
    c = a[t];
    a[t] = (tolower(c));
    t++;
  }
  for (int i = 0; i < a.length(); i++) {
    if ((a[i] == 'a') || (a[i] == 'e') || (a[i] == 'i') || (a[i] == 'o') ||
        (a[i] == 'u') || (a[i] == 'y'))
      cout << "";
    else
      cout << "." << a[i];
  }
}
