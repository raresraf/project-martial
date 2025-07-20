#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdio.h>
int compare(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

using namespace std;
int main() {
  char str[101];
  cin >> str;
  int ar[51], j = 0;
  for (int i = 0; i < strlen(str); i += 2) {
    ar[j] = str[i];
    j++;
  }
  qsort(ar, j + 1, sizeof(int), compare);
  for (int i = 0; i <= j - 1; i++)
    //  cout<<ar[i]<<"  "<<i<<"  "<<j<<endl;
    if (i == j - 1)
      cout << ar[i] - 48;
    else
      cout << ar[i] - 48 << '+';
  return 0;
}
