#include <bits/stdc++.h>
using namespace std;
int main() {
  char ch[101], ch2[101];
  int i, sum = 0, sum1 = 0, sum2 = 0;
  cin >> ch >> ch2;
  for (i = 0; i < strlen(ch); i++) {
    if (ch[i] >= 97 && ch[i] <= 122) {
      ch[i] -= 32;
    }
    if (ch2[i] >= 97 && ch2[i] <= 122) {
      ch2[i] -= 32;
    }

    if (ch[i] > ch2[i]) {
      cout << "1" << endl;
      break;
    } else if (ch[i] < ch2[i]) {
      cout << "-1" << endl;
      break;
    }
  }
  if (ch[i] == ch2[i])
    cout << "0" << endl;
}
