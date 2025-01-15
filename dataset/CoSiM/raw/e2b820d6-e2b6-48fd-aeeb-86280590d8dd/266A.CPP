#include <iostream>
#include <stdio.h>
#include <string.h>
using namespace std;
int main()

{
  char ch[50];
  int count = 0, num;
  // freopen("in.txt","r",stdin);
  cin >> num >> ch;
  for (int i = 0; i < num - 1; i++) {
    if (ch[i] == ch[i + 1]) {
      count++;
    }
  }
  cout << count;
  return 0;
}
