#include <bits/stdc++.h>
using namespace std;
void swap(char *s1, char *s2) {
  char temp = *s1;
  *s1 = *s2;
  *s2 = temp;
}
int main() {
  int n, t, i;
  char a[101];
  cin >> n >> t;
  cin >> a;
  while (t--) {
    for (i = 0; i < (n - 1); i++)
      if (a[i] == 'B' && a[i + 1] == 'G')
        swap(a[i], a[i + 1]), i++;
  }
  cout << a;
}
