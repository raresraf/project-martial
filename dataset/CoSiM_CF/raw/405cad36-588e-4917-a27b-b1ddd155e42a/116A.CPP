//
// Created by Taewoo Kim on 3/23/2017.
//

#include <iostream>

using namespace std;

// only positive numbers
int readInt() {
  char ch = getchar();
  while (true) {
    if (ch >= '0' && ch <= '9')
      break;
    ch = getchar();
  }
  int result = ch - '0';
  while (true) {
    ch = getchar();
    if (ch < '0' || ch > '9')
      break;
    result = result * 10 + ch - '0';
  }
  return result;
}

int main() {
  // find max people at any given time (people exit before others enter)
  int max = 0, count = 0;
  int n = readInt();
  for (int i = 0; i < n; i++) {
    int a = readInt();
    int b = readInt();
    // people get off first, then get on
    count += b - a;
    if (count > max)
      max = count;
  }

  cout << max;
  return 0;
}