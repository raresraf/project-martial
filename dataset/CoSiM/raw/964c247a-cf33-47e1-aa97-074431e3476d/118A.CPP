#include <bits/stdc++.h>
using namespace std;
int main() {
  char ch[100];
  int i;
  scanf("%s", &ch);

  for (i = 0; i < strlen(ch); i++) {
    if (ch[i] >= 65 && ch[i] <= 90) {
      ch[i] = ch[i] + 32;
    }
    if (ch[i] != 'a' && ch[i] != 'e' && ch[i] != 'i' && ch[i] != 'o' &&
        ch[i] != 'u' && ch[i] != 'y') {
      cout << "." << ch[i];
    }
  }
  cout << endl;
}
