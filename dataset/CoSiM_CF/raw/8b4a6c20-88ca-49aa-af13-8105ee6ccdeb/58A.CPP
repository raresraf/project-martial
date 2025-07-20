#include <iostream>
using namespace std;
int main() {
  int i, count, l;
  count = 0;
  l = 0;
  char msg[100];
  cin >> msg;
  while (msg[l] != '\0')
    l++;
  for (i = 0; i < l; i++) {
    if (count == 0) {
      if (msg[i] == 'h')
        count++;
    } else if (count == 1) {
      if (msg[i] == 'e')
        count++;
    } else if (count == 2) {
      if (msg[i] == 'l')
        count++;
    } else if (count == 3) {
      if (msg[i] == 'l')
        count++;
    } else if (count == 4) {
      if (msg[i] == 'o')
        count++;
    } else
      break;
  }
  if (count == 5)
    cout << "YES";
  else
    cout << "NO";
  cout << endl;
  return 0;
}
