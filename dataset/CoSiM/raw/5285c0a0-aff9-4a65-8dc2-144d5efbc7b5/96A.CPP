#include <iostream>
using namespace std;
int check70(char a[]) {
  int flag = 0;
  int count = 0;
  int i;
  while (a[count] != '\0') {
    if (a[count] == '0' && a[count + 1] == '0' && a[count + 2] == '0' &&
        a[count + 3] == '0' && a[count + 4] == '0' && a[count + 5] == '0' &&
        a[count + 6] == '0') {
      flag = 1;
      break;
    } else
      count = count + 1;
  }
  return flag;
}
int check71(char a[]) {
  int flag = 0;
  int count = 0;
  int i;
  while (a[count] != '\0') {
    if (a[count] == '1' && a[count + 1] == '1' && a[count + 2] == '1' &&
        a[count + 3] == '1' && a[count + 4] == '1' && a[count + 5] == '1' &&
        a[count + 6] == '1') {
      flag = 1;
      break;
    } else
      count = count + 1;
  }
  return flag;
}

int main() {
  char q[100];
  cin >> q;
  int flag = 0;
  flag = check70(q);
  if (flag == 1)
    cout << "YES" << endl;
  else {
    flag = check71(q);
    if (flag == 1)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }
  return 0;
}
