#include <bits/stdc++.h>
using namespace std;
int main() {
  char ch[1002];
  cin >> ch;
  for (int i = 0; i < strlen(ch); i++) {
    if ((i == 0) && (ch[i] >= 97 && ch[i] <= 122))
      ch[i] = ch[i] - 32;
  }
  cout << ch << endl;
}
