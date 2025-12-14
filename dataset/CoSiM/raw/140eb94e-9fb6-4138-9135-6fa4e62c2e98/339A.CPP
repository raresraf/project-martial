#include <iostream>
using namespace std;
int strlength(char a[]) {
  int count = 0;
  while (a[count] != '\0')
    count++;
  return count;
}
int main() {
  char a[100];
  char temp;
  cin >> a;
  int i, j, len;
  len = strlength(a);
  //	cout << len <<endl;
  for (i = 2; i < len; i = i + 2) {
    // cout << a[i] << endl;
    for (j = 0; j < i; j = j + 2) {
      // cout << a[j] << endl;
      if (a[i] < a[j]) {
        temp = a[i];
        a[i] = a[j];
        a[j] = temp;
        // cout << a[j] << a[i] << endl;
      }
    }
  }
  cout << a << endl;
  return 0;
}
