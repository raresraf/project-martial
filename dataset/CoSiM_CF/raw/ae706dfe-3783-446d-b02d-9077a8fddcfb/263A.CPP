#include <iostream>
using namespace std;
int main() {
  int a[5][5];
  int id, jd;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        id = i;
        jd = j;
      }
    }
  }
  // We have to make it 2,2
  int cnt = 0;
  if (id > 2) {
    while (id != 2) {
      cnt++;
      id--;
    }
  } else {
    while (id != 2) {
      cnt++;
      id++;
    }
  }
  if (jd > 2) {
    while (jd != 2) {
      cnt++;
      jd--;
    }
  } else {
    while (jd != 2) {
      cnt++;
      jd++;
    }
  }
  cout << cnt;
}
