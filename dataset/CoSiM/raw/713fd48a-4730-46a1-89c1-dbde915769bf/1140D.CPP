#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

void Dump(const std::vector<int>& vec) {
    for (int i = 0; i < (int)vec.size(); ++i) {
        // printf("[%d] %d\n", i, vec[i]);
        printf("%d\n", vec[i]);
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int sum = 0;
    for (int i = 2; i <= n - 1; ++i) {
        sum += i * (i + 1);
    }

    printf("%d\n", sum);

    return 0;
}
