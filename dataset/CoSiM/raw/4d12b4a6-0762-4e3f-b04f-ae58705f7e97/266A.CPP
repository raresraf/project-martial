#include <bits/stdc++.h>

using namespace std;

int main() {
  char a[101];
  int i, n, j = 0, count = 0;
  scanf("%d", &n);
  for (i = 1; i <= n; i++) {
    scanf("%s", &a[i]); //%s bisa utk string n char ,,  kalau pake %c maka beri
                        //spasi belakang " %c"
  }                     // utk char, beri simbol &
  a[j] = 'a';
  for (i = 1; i <= n; i++) {
    if (a[i] == a[j]) {
      count++;
      j++;
    } else {
      j++;
    }
  }
  cout << count << endl;
}
