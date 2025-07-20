#include <iostream>
using namespace std;
void bsort(int ara[], int n);
int main() {
  string s;
  int i, j, ara[100], cnt = 0;
  cin >> s;
  for (i = 0; i < s.length(); i++) {
    if (s[i] >= 48 && s[i] <= 57)
      ara[cnt++] = s[i] - 48;
  }
  bsort(ara, cnt);
  for (i = 0; i < cnt - 1; i++)
    cout << ara[i] << "+";
  cout << ara[cnt - 1] << endl;
  return 0;
}
void bsort(int ara[], int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (ara[j] > ara[j + 1])
        swap(ara[j], ara[j + 1]);
    }
  }
}
