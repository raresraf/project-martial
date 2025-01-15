#include <bits/stdc++.h>
using namespace std;
int main() {
  string ab;
  int arr[30];
  int m = 0;
  cin >> ab;
  for (int i = 0; ab[i] != '\0'; i++) {
    if ((ab[i] == 'a') || (ab[i] == 'b') || (ab[i] == 'c') || (ab[i] == 'd') ||
        (ab[i] == 'e') || (ab[i] == 'f') || (ab[i] == 'g') ||
        (ab[i] == 'h' || ab[i] == 'H') || (ab[i] == 'i' || ab[i] == 'I') ||
        (ab[i] == 'j') || (ab[i] == 'k') || (ab[i] == 'l') || (ab[i] == 'm') ||
        (ab[i] == 'n') || (ab[i] == 'o') || (ab[i] == 'p') || (ab[i] == 'q') ||
        (ab[i] == 'r') || (ab[i] == 's') || (ab[i] == 't') || (ab[i] == 'u') ||
        (ab[i] == 'v') || (ab[i] == 'w') || (ab[i] == 'x') || (ab[i] == 'y') ||
        (ab[i] == 'z')) {
      arr[int(ab[i]) - 97] = ab[i];
    }
  }
  // cout<<m<<endl;
  for (int i = 0; i < 26; i++) {
    if (arr[i] >= 97 && arr[i] <= 122) {

      m++;
    }
  }
  // cout<<m<<endl;
  if ((m % 2) == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
}