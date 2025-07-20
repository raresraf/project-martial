#include <bits/stdc++.h>
using namespace std;
int main() {
  char arr[100], x[200];
  cin >> arr;
  int l = strlen(arr), c = 0, i;
  arr[l] = '\0';
  for (i = 0; i < l; i++) {
    if (arr[i] != 'A' && arr[i] != 'a' && arr[i] != 'E' && arr[i] != 'e' &&
        arr[i] != 'I' && arr[i] != 'i' && arr[i] != 'O' && arr[i] != 'o' &&
        arr[i] != 'U' && arr[i] != 'u' && arr[i] != 'Y' && arr[i] != 'y') {
      x[c] = arr[i];
      c++;
    }
  }
  for (i = 0; i < c; i++) {
    if (isupper(x[i]))
      x[i] = tolower(x[i]);
  }
  for (i = 0; i < c; i++)
    cout << "." << x[i];
  cout << endl;
  return 0;
}