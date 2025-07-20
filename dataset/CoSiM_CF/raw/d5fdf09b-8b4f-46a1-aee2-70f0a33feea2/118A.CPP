#include <bits/stdc++.h>
using namespace std;

int main() {
  char a[106], final[300];
  cin >> a;

  for (int i = 0; i < strlen(a); i++)
    a[i] = tolower(a[i]);

  int k = 0;

  for (int i = 0; i < strlen(a); i++) {
    char ch = a[i];
    if (ch != 'a' && ch != 'e' && ch != 'i' && ch != 'o' && ch != 'u' &&
        ch != 'y') {
      final[k] = '.';
      k++;
      final[k] = ch;
      k++;
    }
  }
  final[k] = '\0';
  cout << final;

  return 0;
}
