#include <bits/stdc++.h>
using namespace std;

int main() {
  char s[100];
  cin >> s;
  int H = 0, E = 0, L = 0, O = 0;
  int n = strlen(s);
  for (int i = 0; i < n; i++) {
    if (H == 0 && E == 0 && L == 0 && O == 0 && s[i] == 'h')
      H++;
    else if (H == 1 && E == 0 && L == 0 && O == 0 && s[i] == 'e')
      E++;
    else if (H == 1 && E == 1 && L < 2 && O == 0 && s[i] == 'l')
      L++;
    else if (H == 1 && E == 1 && L == 2 && O == 0 && s[i] == 'o')
      O++;
  }
  if (H == 1 && E == 1 && L == 2 && O == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
