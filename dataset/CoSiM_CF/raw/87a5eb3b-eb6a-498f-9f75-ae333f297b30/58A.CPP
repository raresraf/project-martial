#include <bits/stdc++.h>

using namespace std;

bool h = false, e = false, l1 = false, l2 = false, o = false;

int main() {
  char slovo[101];
  cin >> slovo;

  for (int i(0); i < strlen(slovo); i++) {
    if (slovo[i] == 'h' && !h && !e && !l1 && !l2 && !o) {
      h = true;
    }
    if (slovo[i] == 'e' && h && !e && !l1 && !l2 && !o) {
      e = true;
    }
    if (slovo[i] == 'l' && h && e && !l1 && !l2 && !o) {
      l1 = true;
    } else if (slovo[i] == 'l' && h && e && l1 && !l2 && !o) {
      l2 = true;
    }
    if (slovo[i] == 'o' && h && e && l1 && l2 && !o) {
      o = true;
    }
  }
  if (h && e && l1 && l2 && o) {
    cout << "YES" << '\n';
  } else {
    cout << "NO" << '\n';
  }
  return 0;
}
