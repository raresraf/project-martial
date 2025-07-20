#include <cstring>
#include <iostream>
#include <stdlib.h>

int compare(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

int main() {
  char str[105];
  int array[100], i, j;
  std::cin >> str;
  j = 0;
  for (i = 0; i < strlen(str); i = i + 2) {
    array[j] = (int)str[i] - 48;
    j++;
  }
  qsort(array, j, sizeof(int), compare);
  // printing the new string
  for (i = 0; i < j - 1; i++)
    std::cout << array[i] << "+";
  std::cout << array[j - 1] << '\n';
  return 0;
}
