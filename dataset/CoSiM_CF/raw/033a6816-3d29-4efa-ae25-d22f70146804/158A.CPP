#include <iostream>
#include <stdlib.h>
using namespace std;
int main() {
  int total = 0, min = 0, k = 0, temp = 0, score = 0;
  cin >> total;
  cin >> min;
  k = 0;
  int *student_list = (int *)malloc(total * sizeof(int));
  while (k < total) {
    cin >> student_list[k++];
  }
  min = student_list[min - 1];
  k = 0;
  while (k < total) {
    if (student_list[k] >= min && student_list[k] != 0) {
      score++;
    }
    k++;
  }
  cout << score;
  return 0;
}
